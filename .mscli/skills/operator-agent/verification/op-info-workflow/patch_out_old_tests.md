<a id="patch-out-old-tests-steps"></a>
`op_database.py` contains both existing test cases and newly added ones. To test only the newly added cases, create a new `xxx_db` list that contains only the new APIs and override, rather than delete, the existing database registration.

Do not use any lint tool to reformat code in this step.

Example:
```
@@ -7848,6 +7969,7 @@ other_op_db = [
     'mint.nn.functional.linear',
     'mint.nn.Linear',
     'mint.nn.Conv1d',
+    'mint.nn.Conv2d',
     'Tensor.masked_scatter',
     'Tensor.masked_scatter_',
     'Tensor.add_',
```

This commit adds a test for the `mint.nn.Conv2d` API to `other_op_db`. To override the existing registration, create a new `other_op_db` after the original registration:

```
other_op_db = [
    ...
]
other_op_db = ['mint.nn.Conv2d']
```

This overrides the existing registration so that subsequent test validation runs only the newly added API.

For `xxx_db` lists with no newly added APIs, override them with an empty list:

```
binary_op_db = [
    ...
]
binary_op_db = []
```

Finally, make sure every `xxx_db` has been handled. After the modification is complete, create a git commit with the message `op-info-test: patch out old tests`.
